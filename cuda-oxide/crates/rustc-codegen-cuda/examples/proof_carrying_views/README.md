# Proof-carrying views

This example compares safe GPU views with equivalent raw-pointer kernels.
The safe path proves a thread's complete range before access, then carries that
proof into each load and store. It keeps the same 32-bit coordinate math and
memory operations as the raw path.

```text
ordinary safe slice:  bounds check -> load -> bounds check -> store
proof-carrying view:  one range check -> [load, store] with the proof
raw pointer:          manual guards -> [load, store] in unsafe code
```

Three pairs are compiled:

- one element per thread, through `InBoundsMut32`
- four adjacent elements per thread, through `StaticViewMut32<T, 4>`
- a two-column row-major GEMM epilogue tile, through
  `StaticTileMut32<T, 1, 2, 64>`

The example also pins the legacy safety fix for a wrong-rank launch:

```text
Before: (x=0, y=0) -> 0 and (x=0, y=1) -> 0  (alias)
After:  non-unit Y/Z dimensions -> invalid witness -> no mutable access
```

The launch contract proves that every active coordinate axis fits in `u32`.
`thread::index_1d_u32(launch_context)` and
`thread::coord_2d_u32(launch_context)` may therefore use 32-bit arithmetic
without truncation. The host validates this once while preparing the launch.

```text
#[kernel(launch_context = launch_context)] -> entry creates checked context
thread::index_1d_u32(launch_context)       -> returns this thread's 1-D index
```

The launch context is explicit so the call shows where its checked launch facts
come from. It is a local device value, not a kernel argument.

The 2-D epilogue encodes its logical row width in
`RowMajorTiles<1, 2, 64>`. The caller must choose `64` to match the buffer's
actual layout. One check proves the full rectangle stays within its row and
parent allocation; both interior accesses are then check-free.

This first view layer deliberately exposes a checked boundary and natural `T`
alignment only. It does not label an edge tile as exact, padded, or masked, and
it does not claim stronger vector alignment without a separate proof.

```bash
cargo oxide build proof_carrying_views
cuda-oxide/crates/rustc-codegen-cuda/examples/proof_carrying_views/target/release/proof_carrying_views --verify-ptx
```

The verifier needs no GPU. It checks that the safe and raw entries keep the
same global-memory load/store operations, use 32-bit coordinate math, and
contain no compile-time contract calls. Both element and tile access compile to
one outer guard branch; all four tile loads and stores run without another
check. It also checks that the legacy 1-D path reads and validates every Y/Z
grid and block dimension before storing.

This is entry-body parity, not whole-module size parity. The current exporter
can retain inlined, uncalled helpers as `.func` definitions; they do not run on
the hot path. Internalizing and dead-stripping those helpers is separate work.
