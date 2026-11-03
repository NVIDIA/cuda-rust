# enum_state_transitions

End-to-end device-code conformance coverage for state transitions in
`Option<u32>` and combinator-driven variant transitions in `Result<u32, u32>`.

## What this tests

The example deliberately chooses runtime-selected starting variants so enum
construction, discriminant reads, payload extraction, and payload mutation stay
observable in both optimized and low-MIR-optimization builds.

### `Option`

The `option_transitions` kernel covers both `Some` and `None` starting states for:

- `Option::take`
- `Option::replace`
- `Option::get_or_insert`

The host verifies both returned values and final enum states. The
`get_or_insert` result is also mutated through the returned `&mut` payload,
covering the projected payload-address path.

### `Result`

The `result_combinators` kernel covers both `Ok` and `Err` starting states for:

- `Result::map`
- `Result::map_err`
- `Result::and_then`
- `Result::or_else`

The cases deliberately include both variant-preserving and variant-changing
operations:

```text
Ok  -> Ok
Ok  -> Err
Err -> Err
Err -> Ok
```

Every intermediate result is matched and packed into an integer observed by the
host, forcing the final states to exercise discriminant reads and payload
extraction rather than only checking that the methods compile.

The device path uses no allocator-backed data structures.

## Usage

```bash
cargo oxide run enum_state_transitions
CUDA_OXIDE_NO_OPT=1 cargo oxide run enum_state_transitions
```

## Expected output

```text
=== enum_state_transitions ===
PASS: Option::take
PASS: Option::replace
PASS: Option::get_or_insert
PASS: Result::map
PASS: Result::map_err
PASS: Result::and_then
PASS: Result::or_else
PASS: enum_state_transitions
```
