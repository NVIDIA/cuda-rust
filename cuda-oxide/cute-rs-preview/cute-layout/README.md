# Layout algebra for cuda-oxide

`cute-layout` describes and evaluates coordinate-to-memory mappings. It is a
`no_std` crate using `alloc`, shared by the Rust tensor API and the compiler.

```rust
use cute_layout::{IntTuple, Layout};

let layout: Layout = "(2,3):(3,1)".parse().unwrap();
assert_eq!(layout.checked_call(&IntTuple::Leaf(1)), Some(3));
```

## Module navigation

| Module | Responsibility |
| --- | --- |
| [`tuple`](src/tuple.rs) | Nested `IntTuple` shapes, strides and coordinates |
| [`layout`](src/layout.rs) | `Layout` construction and coordinate evaluation |
| [`parse`](src/parse.rs) | Text formatting, parsing and `ParseLayoutError` |
| [`algebra::mapping`](src/algebra/mapping.rs) | Coalescing, composition, complements, inverse maps and reshaping |
| [`algebra::tiling`](src/algebra/tiling.rs) | Logical division, blocked products and raked products |
| [`algebra::thread_value`](src/algebra/thread_value.rs) | Thread/value ownership maps |
| [`algebra::alignment`](src/algebra/alignment.rs) | Static vector-access alignment |
| [`composed`](src/composed.rs) and [`swizzle`](src/swizzle.rs) | Composed addresses, offset units and shared-memory swizzles |
| [`predication`](src/predication.rs) | Safe source selection for partial copy atoms |
| [`validation`](src/validation/mod.rs) | Bounded validation for thread/value layouts, cooperative copies, TMA and ldmatrix |

The root imports `IntTuple`, `Layout`, `ParseLayoutError` and `crd2idx` remain
available. Every existing `algebra::*` function remains available through an
explicit reexport. Grouped imports such as `cute_layout::layout::Layout` and
`cute_layout::algebra::mapping::composition` refer to the same definitions.

Use checked evaluation and the validation modules for layouts decoded from
external compiler input. The unchecked constructors and algebra retain their
documented arithmetic and shape preconditions.

## Validation

Run from `cuda-oxide/` with its pinned nightly:

```sh
cargo test -p cute-layout
cargo clippy -p cute-layout --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p cute-layout --no-deps
cargo fmt -p cute-layout --check
```

Tests cover the defining algebra equations, checked arithmetic, representation,
parsing, static validation, and interoperability of the grouped and existing
public paths. Algebra contract tests live beside their shared fixtures under
[`src/algebra/tests`](src/algebra/tests).
