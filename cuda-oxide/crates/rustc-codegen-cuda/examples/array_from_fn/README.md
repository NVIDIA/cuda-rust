# `array::from_fn` symbol-collision repro

```bash
cargo oxide run array_from_fn --arch sm_120
cargo oxide run array_from_fn --emit-nvvm-ir --arch sm_120
```

A single generic `#[device]` helper calls `core::array::from_fn` with a
capturing closure. The kernel instantiates that helper at lengths **1 and 4**
and checks all five elements across 64 lanes.

Before the fix, LLVM export reports that two `core::array::try_from_fn`
instances normalize to the same `@make_array...` name. The device marker
occurs inside the closure type's mangled parent path. Removing everything
through that marker also removes the outer function identity and array length.
Preserving Rust-mangled names keeps the instances distinct in definitions,
direct calls, and function addresses. Plain device names still lose their
internal prefix; device extern names retain their existing behavior.

The exporter regression uses the original rejected bending-action symbols and
checks both modern and legacy NVVM syntax without a GPU:

```bash
cargo test -p llvm-export mangled_array_from_fn_specializations_keep_distinct_definitions_and_references
```
