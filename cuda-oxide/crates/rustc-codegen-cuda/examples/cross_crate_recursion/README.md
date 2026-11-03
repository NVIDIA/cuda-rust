# cross_crate_recursion

Regression example for cross-crate recursive device functions.

```
cargo oxide run cross_crate_recursion   # expects: PASS cross_crate_recursion: sum=136
```

The kernel calls `reprolib::rec1`, a recursive function in a dependency crate. Without
`-Zalways-encode-mir` in cargo-oxide's device rustflags, device codegen emitted a call to
`reprolib__rec1` with no definition and failed module verification with `Symbol reprolib__rec1 not
found`. Defining the same `rec1` locally in this crate always worked — isolating the trigger to
"cross-crate + not inlinable" (recursion is the canonical un-inlinable case).
