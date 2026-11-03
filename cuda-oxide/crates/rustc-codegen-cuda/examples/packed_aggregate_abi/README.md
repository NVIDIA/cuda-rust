# packed_aggregate_abi

End-to-end regression coverage for packed aggregate lowering and CUDA ABI handling.

The example validates:

- `#[repr(C, packed)]` kernel parameters passed by value;
- `#[repr(C, packed(2))]` kernel parameters passed by value;
- internal device helper argument/return paths for packed aggregates;
- whole-value packed aggregate stores and loads;
- exact host size/alignment/field offsets;
- PTX aggregate parameter byte sizes and ABI alignments;
- stored field bytes, while deliberately ignoring unspecified Rust padding bytes.

Expected host layouts:

- `Packed1`: size 5, alignment 1, `b` at byte offset 1;
- `Packed2`: size 6, alignment 2, `b` at byte offset 2.

Run through the normal cuda-oxide example workflow. Once the generated PTX exists,
its parameter shapes can also be checked without launching CUDA:

```text
cargo run -- --verify-ptx
```
