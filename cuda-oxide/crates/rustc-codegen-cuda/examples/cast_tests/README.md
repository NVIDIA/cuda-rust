# Cast Operations Test

Comprehensive test suite for MIR cast kind handling in the codegen backend. Exercises every
`CastKind` variant that can appear in device code, including the struct↔ptr conversions
required for fat pointers and newtypes.

## Running

```bash
cargo oxide run cast_tests
```

## Test Matrix

| Category                                | Tests | CastKind / Feature                         | LLVM lowering                        |
|-----------------------------------------|-------|--------------------------------------------|--------------------------------------|
| IntToInt (zext, sext, trunc)            |     3 | `IntToInt`                                 | `sext`, `zext`, `trunc`              |
| IntToFloat (u32→f32, i32→f32)           |     2 | `IntToFloat`                               | `uitofp`, `sitofp`                   |
| FloatToInt (f32/f64 → i32/u32/i64/u64)  |     8 | `FloatToInt`                               | `llvm.fptosi.sat`, `llvm.fptoui.sat` |
| FloatToFloat (fpext, fptrunc)           |     2 | `FloatToFloat`                             | `fpext`, `fptrunc`                   |
| bool → u32                              |     1 | `IntToInt`                                 | `zext`                               |
| Transmute (i32↔f32, u64↔f64, u32↔i32)   |     4 | `Transmute`                                | `bitcast` / `extractvalue`           |
| Safe bit ops (from_bits, to_bits, etc.) |     5 | `Transmute` / `IntToInt`                   | `bitcast`                            |
| PtrToPtr, ptr→usize                     |     2 | `PtrToPtr`, `PointerExposeProvenance`      | `bitcast`, `ptrtoint`                |
| ConstantIndex on slice                  |     1 | (projection handling, not a cast)          | `extractvalue` + `GEP` + `load`      |
| Unsize f32 (&[T;N]→&[T], iter sum)      |     3 | `PointerCoercion(Unsize)`                  | `extractvalue` / `insertvalue`       |
| Unsize f64 (as_slice, explicit, iter)   |     3 | `PointerCoercion(Unsize)`                  | `extractvalue` / `insertvalue`       |
| **Total**                               | **34**|                                            |                                      |

FloatToInt covers f32→u32, f32→i32, f64→u32, f64→i32, f64→u64, f64→i64, and mixed precision (f32→u64, f32→i64). All use saturating semantics (Rust-defined).

## Architecture

All casts dispatch on `MirCastKindAttr` — a pliron attribute preserved from Rust MIR — rather
than guessing semantics from source/destination types.

```text
Rust MIR                    dialect-mir                  LLVM dialect
──────────                  ───────────                  ────────────
Rvalue::Cast         ──►    MirCastOp                ──► Specific LLVM
(CastKind,                  + MirCastKindAttr            cast instruction
 operand, ty)               (semantic intent)

mir-importer/               dialect-mir/                 mir-lower/
rvalue/expr.rs              ops/cast.rs                  convert/ops/cast.rs
                            attributes.rs
```

Pointer-related casts (`Transmute`, `PtrToPtr`, `FnPtrToPtr`, all `PointerCoercion*`, `Subtype`)
go through `emit_pointer_cast`, which handles struct↔ptr conversions generically:

| Source → Dest          | LLVM Operation         | Example                        |
|------------------------|------------------------|--------------------------------|
| struct → ptr           | `extractvalue` field 0 | `{ptr, i64}` → `ptr` (slice)   |
| ptr → struct           | `insertvalue` undef    | `ptr` → `{ptr}` (NonNull)      |
| ptr → ptr (diff AS)    | `addrspacecast`        | generic → shared memory        |
| otherwise              | `bitcast`              | `*mut T` → `*const T`          |

## Expected Output

```text
=== Cast Operations Test Suite ===

--- Numeric Casts (IntToInt, IntToFloat, FloatToInt, FloatToFloat) ---
  [PASS] u32 → u64: 42 → 42
  [PASS] u64 → u32 (trunc): 0x10000002A → 42
  [PASS] i32 → i64 (sext): -7 → -7
  [PASS] u32 → f32: 42 → 42
  [PASS] i32 → f32: -7 → -7
  [PASS] f32 → u32: 42.9 → 42
  [PASS] f32 → i32: -7.8 → -7
  [PASS] f64 → u32: 42.9 → 42
  [PASS] f64 → i32: -7.8 → -7
  [PASS] f64 → u64: 100.5 → 100
  [PASS] f64 → i64: -100.5 → -100
  [PASS] f32 → u64 (mixed): 42.9 → 42
  [PASS] f32 → i64 (mixed): -7.8 → -7
  [PASS] f32 → f64: 3.14 → 3.140000104904175
  [PASS] f64 → f32: 3.14 → 3.14
  [PASS] bool → u32: true → 1

--- Transmute (bit reinterpretation) ---
  [PASS] transmute i32(0x3F800000) → f32: 1
  [PASS] transmute f32(1.0) → u32: 0x3F800000
  [PASS] transmute u64(0x4000...) → f64: 2
  [PASS] transmute u32(0xFFFFFFFF) → i32: -1

--- Safe Bit Reinterpretation (from_bits, to_bits, cast_signed, cast_unsigned) ---
  [PASS] f32::from_bits(i32::cast_unsigned(0x3F800000)): 1
  [PASS] f32::to_bits(1.0): 0x3F800000
  [PASS] f64::from_bits(0x4000...): 2
  [PASS] u32::cast_signed(0xFFFFFFFF): -1
  [PASS] i32::cast_unsigned(-1): 0xFFFFFFFF

--- Pointer Casts (PtrToPtr, PointerExposeProvenance) ---
  [PASS] ptr reinterpret: got address 0x... (non-null)
  [PASS] ptr → usize: got address 0x... (non-null)

--- ConstantIndex on Slice (Bug 2) ---
  [PASS] slice constant index: data[0] = 42

--- Unsizing Coercions (&[T; N] → &[T]) ---
  [PASS] array → slice: [10.0, 20.0, 30.0, 40.0]
  [PASS] array.as_slice(): [100.0, 200.0, 300.0, 400.0]
  [PASS] array iter sum: 6

--- f64 Array Unsizing ---
  [PASS] f64 as_slice: [1.0, 2.0, 3.0]
  [PASS] f64 explicit slice: [1.0, 2.0, 3.0]
  [PASS] f64 iter sum: 6

=========================================
RESULTS: 34 passed, 0 failed
=========================================
```
