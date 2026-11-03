# error_set_discriminant_uninhabited

Negative test: confirms that cuda-oxide rejects MIR `SetDiscriminant` when
the requested variant cannot contain a value.

```rust
enum Never {}
enum State { First, Impossible(Never), Last }

SetDiscriminant(state, Impossible); // must reject, never write tag 1
```

The enum still has a normal direct tag because `First` and `Last` are both
inhabited. A compiler that blindly writes the requested tag would return an
invalid Rust value, so this case must fail before LLVM is generated.

## Usage

```bash
cargo oxide run error_set_discriminant_uninhabited
```

The build must fail with:

```text
SetDiscriminant cannot select uninhabited variant 1
```

`cuda-oxide/scripts/smoketest.sh` classifies this as a diagnostics fixture: rejection is
the correct behavior, not a missing supported operation.
