# drop_glue

Positive device-side conformance test for drop execution and suppression
semantics.

## What this tests

rustc emits `TerminatorKind::Drop` for places whose type has drop glue
(non-`Copy` types with an `impl Drop`, recursively through fields and
parameters). cuda-oxide translates effectful drops into device-side
`drop_in_place` calls so destructors run on the GPU.

The historical regression remains intact: a `DropMarker` writes
`0xDEADBEEF` through a captured pointer when it leaves scope, and the host
verifies that this happens for all 256 threads.

The example also covers the core initialization/drop wrappers that either
suppress or explicitly trigger destruction:

- ordinary scope exit runs `Drop`;
- `ManuallyDrop::new` suppresses automatic destruction;
- `ManuallyDrop::drop` explicitly runs the contained destructor;
- `mem::forget` consumes a value without running `Drop`;
- dropping `MaybeUninit<T>` does not drop an initialized `T`;
- `MaybeUninit::write` followed by `assume_init_drop` explicitly drops `T`;
- `MaybeUninit::assume_init_read` produces an owned `T` that subsequently
  follows normal drop semantics.

The focused kernel uses a `DropMarker` whose destructor writes
`0xDEADBEEF`. Each lane starts with a distinct value, allowing the host to
distinguish a destructor that ran from one that was correctly suppressed.

## Coverage matrix

| Lane | Operation | Expected result |
| --- | --- | --- |
| 0 | ordinary scope exit | `0xDEADBEEF` |
| 1 | `ManuallyDrop::new` | `0x22220000` |
| 2 | `ManuallyDrop::drop` | `0xDEADBEEF` |
| 3 | `mem::forget` | `0x44440000` |
| 4 | `MaybeUninit::new` leaving scope | `0x55550000` |
| 5 | `MaybeUninit::write` + `assume_init_drop` | `0xDEADBEEF` |
| 6 | `MaybeUninit::assume_init_read` | `0xDEADBEEF` |

## Usage

Optimized:

```bash
cargo oxide run drop_glue
```

Low MIR optimization:

```bash
CUDA_OXIDE_NO_OPT=1 cargo oxide run drop_glue
```

## Expected output

```text
=== drop_glue ===

SUCCESS: drop glue wrote sentinel in all 256 elements
PASS: ordinary scope exit runs drop glue
PASS: ManuallyDrop suppresses automatic drop
PASS: ManuallyDrop::drop runs drop glue
PASS: mem::forget suppresses drop
PASS: MaybeUninit suppresses contained drop
PASS: MaybeUninit::assume_init_drop runs drop glue
PASS: MaybeUninit::assume_init_read preserves inhabited drop path
PASS: initialization/drop conformance
```
