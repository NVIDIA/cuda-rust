# float_rounding

## Rounding without libdevice

This example demonstrates that kernels whose only "math library" usage is
rounding (`floor`, `ceil`, `trunc`, `round`, `round_ties_even`) compile on
the self-contained PTX path: the rounding methods lower to the native LLVM
intrinsics (`llvm.floor.*`, `llvm.ceil.*`, `llvm.trunc.*`, `llvm.round.*`,
`llvm.roundeven.*`) instead of libdevice `__nv_*` calls, and the NVPTX
backend selects a single `cvt` instruction for each (plain `round` becomes
a short inline sequence, since the hardware has no ties-away-from-zero
`cvt` mode).

## What This Example Does

- Two kernels — one per float width — evaluate all five rounding ops per
  input element and write the raw IEEE-754 bits.
- The host compares every result bit-exactly against the host stdlib and
  additionally pins the named halfway/sign cases against literals:
  `round(2.5) == 3.0`, `round(-2.5) == -3.0`,
  `round_ties_even(2.5) == 2.0`, `round_ties_even(3.5) == 4.0`,
  `round(-0.4) == -0.0` (signed zero), `floor(-1.5) == -2.0`,
  `ceil(-1.5) == -1.0`, `trunc(-1.7) == -1.0`.
- The generated `float_rounding.ll` must contain no `__nv_` symbols.

Exits 0 on PASS, 1 on FAIL.

## Pipeline

No libdevice is detected, so cuda-oxide emits PTX directly via `llc` and
the module loads from the embedded PTX bundle. With `--emit-nvvm-ir` the
same source instead routes rounding through libdevice (`__nv_floorf`, ...),
because the legacy LLVM 7-based NVVM IR dialect predates `llvm.roundeven`.

Run:

```bash
cargo oxide run float_rounding
```
