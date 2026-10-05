# constant_index_from_end

Regression example for issue #751: slice suffix patterns such as
`if let [.., last] = *input` were rejected by the mir-importer.

Matching the end of a slice compiles to the MIR place projection
`ConstantIndex { offset, from_end: true }`, meaning "element
`len - offset`". For fixed-size arrays rustc resolves the position at
compile time (`from_end: false`), so only runtime-length slices reach
the importer with `from_end: true`, and the element index must be
computed at runtime from the slice's length.

Before the fix, both the read and write paths bailed out:

```text
Unsupported construct: ConstantIndex with from_end=true not yet supported
```

The fix uses the fat-pointer representation already in place for
slices: a `&[T]` is a (data pointer, length) pair. When a from-end
index immediately follows a fat-slice deref, the address walker
extracts the length from field 1 of the fat value, materializes
`len - offset` with a runtime subtract, and feeds it to the existing
pointer-offset addressing. The MIR pattern-length test dominates the
projection, so the subtraction cannot underflow on an executed path.

The length must come from the fat value itself, never from the data
pointer's pointee type. For `&[[u32; 3]]`, the pointee array length 3
is the row width; the from-end index selects a row from the outer
runtime-length slice, so subtracting from 3 would read the wrong row.
From-end indexes without that fat-deref provenance (e.g. after a
`Subslice`) still fail closed instead of guessing a length.

## Kernels

| Kernel             | Shape pinned                                              |
|:-------------------|:----------------------------------------------------------|
| `read_last`        | `[.., last]` on `&[u32]` (`offset = 1`)                   |
| `read_penultimate` | `[.., penultimate, _]` on `&[u32]` (`offset = 2`)         |
| `read_last_row`    | `[.., row]` on `&[[u32; 3]]` (outer length, not row width) |
| `write_last`       | `[.., ref mut last]` on `&mut [u32]` (address/store path) |

Each kernel writes its result into a zeroed output buffer; the host
reads everything back and checks every lane, so an index computed from
the wrong length (or a write that lands in a copy) fails loudly. The
harness prints `PASS` when all kernels check out and exits non-zero
otherwise.

## Build & run

From the cuda-oxide repository root:

```bash
cargo oxide run constant_index_from_end
cargo oxide pipeline constant_index_from_end    # dump MIR + LLVM IR
```

Requires a CUDA-capable GPU and the cuda-oxide rustc toolchain.
