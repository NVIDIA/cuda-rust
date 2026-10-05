# set_discriminant

Positive test: verifies that the mir-importer lowers MIR
`StatementKind::SetDiscriminant` to a device-side enum tag write, rather than
silently dropping the discriminant update.

## What this tests

The example calls `custom_mir` helpers from a `#[kernel]` to emit
`StatementKind::SetDiscriminant` directly. It covers both physical layouts
that can be handled without niche encoding:

```text
Direct tag:  Full(7) --write signed i8 tag -5--> Empty
Single/no-tag: Live(7) --no physical tag--> Live(7)
```

The direct-tag enum uses sparse signed discriminants, so the generated store
must contain the declared `-5` bit pattern rather than variant index `0`. The
single-layout enum has an impossible `Never` variant; selecting `Live` must be
a no-op that preserves its payload. Each thread writes `1` only if both checks
pass.

Before the lowering was implemented, this produced:

```
Unsupported construct: SetDiscriminant statements are not yet supported on the device;
until they are lowered, enum discriminant writes would be silently dropped
```

## Usage

```bash
cargo oxide run set_discriminant
```

## Expected output

The build succeeds and the host verifies that every thread observed the
discriminant write:

```
PASS: all 64 threads observed the SetDiscriminant write.
```
