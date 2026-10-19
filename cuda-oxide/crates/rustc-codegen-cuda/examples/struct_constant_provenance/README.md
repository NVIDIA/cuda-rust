# struct_constant_provenance

Positive regression for pointer provenance in direct and nested struct constants.

The example covers two related constant representations:

- a thin reference field that points to a Rust static;
- a slice fat-pointer field whose data word points into a Rust static while its
  second word carries the slice length.

For both forms, the stored pointer bytes contain only the byte addend into the
target allocation. Rustc's provenance table identifies the static allocation
that provides the pointer provenance. The MIR importer must combine those
pieces instead of reconstructing a pointer from placeholder bytes.

The slice regression additionally checks a non-zero target addend and a nested
struct field offset. Its length metadata must remain independent from the data
pointer relocation.

Run with:

```bash
cargo oxide run struct_constant_provenance
```

Expected output:

```text
PASS: struct constant pointer and slice provenance preserved at runtime
```
