# Ref-Operand Mul (issue #133)

Regression test for [#133](https://github.com/NVlabs/cuda-oxide/issues/133):
device codegen rejected `&tmp * &tmp` on a struct whose `Mul` impl lives on
the reference type with

```text
Unsupported construct: Alias type not yet supported:
AliasDef(DefId { id: 22, name: "std::ops::Mul::Output" })
```

## Root cause

The importer typed call results from the callee's declared trait signature.
For a trait-method call, that signature is written against the trait, so its
return type is the unresolved associated-type projection
`<&Foo as Mul>::Output` rather than the concrete type. A name-matching
fallback in the type translator guessed "arithmetic `Output` = self type",
which only accepted non-reference self types, so `impl Mul for &Foo` fell
through to the unsupported-type arm. Worse, the guess itself was wrong for
this shape: `Output` is `Foo` (the owned struct) while `Self` is `&Foo` (a
pointer), so extending the guess would have mistyped the call result.

## Fix

`cuda-oxide/crates/mir-importer/src/translator/terminator/mod.rs` now types call
results from the caller's destination place
(`destination.ty(body.locals())`), which rustc has already monomorphized and
normalized to the concrete type. The callee's `mir.func` return type is
independently derived from the callee body's return place, normalized the
same way, so caller and callee always agree. The name-matching `Output`
guess in `cuda-oxide/crates/mir-importer/src/translator/types.rs` is removed entirely;
projections that somehow still reach the type translator now fail loudly
instead of being guessed.

## Running

```bash
cargo oxide run ref_operand_mul
```

Expected output: `ref_pieces_mul: PASS (result = 289)`.
