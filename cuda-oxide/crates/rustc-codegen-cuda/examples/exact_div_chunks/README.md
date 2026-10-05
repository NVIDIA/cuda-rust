# exact_div_chunks

Exercises `core::intrinsics::exact_div` and `slice::as_chunks`, the safe
chunked-access API that needs it.

Before the intrinsic was implemented, `as_chunks` failed to translate:

```
Translation failed: core::slice::as_chunks::<4>
  [core/src/slice/mod.rs:1345:32] Compilation error: invalid input program
```

Line 1345 is `exact_div(self.len(), N)`.

## Run

```bash
cargo oxide run exact_div_chunks --arch sm_86
```

Expected output:

```
as_chunks::<4>  : 256 / 256 correct
exact_div direct: 256 / 256 correct

PASS
```

## What it checks

`chunk_sum` reads each thread's four elements through `input.as_chunks::<4>()`
and combines them with distinct weights, so a wrong chunk boundary or a
permutation inside a chunk produces a wrong value rather than passing.

`exact_div_direct` calls `core::intrinsics::exact_div` itself (via
`#![feature(core_intrinsics)]`), away from `as_chunks`. The lowering picks
`udiv` or `sdiv` from the operand's signedness, so the kernel covers both
arms: an unsigned `u64` division (`((i + 1) * 256) / 4`) and a signed `i64`
division with a negative dividend (`-((i + 1) * 128) / 4`). Every dividend is
a nonzero exact multiple of its divisor; anything else is undefined behaviour
under the intrinsic's contract.

## Note on codegen

`as_chunks` yields `[f32; 4]` at 4-byte alignment, and the resulting loads stay
scalar rather than fusing into `ld.global.v4`. This example demonstrates that
the safe API now compiles and is correct; access width is governed separately by
the alignment of the element type.
