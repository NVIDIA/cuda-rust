# cuTile Rust Compiler

This crate compiles Rust DSL kernels into Tile IR bytecode for GPU execution
via `tileiras`. Most users interact with it indirectly through `cutile` and
`cutile-macro`.

The runtime resolves `tileiras` in this order:

1. `CUTILE_TILEIRAS_PATH`, when set.
2. `$CUDA_TOOLKIT_PATH/bin/tileiras`, then `$CUDA_HOME/bin/tileiras`, when the
   variable is set and the binary exists there (the same variables, in the
   same order, that the build scripts honor).
3. Standard CUDA 13.3/13.2 install locations, when they contain
   `bin/tileiras`.
4. `tileiras` through normal `PATH` lookup.

The bytecode version handed to `tileiras` is negotiated per toolchain: an
explicit `CUTILE_BYTECODE_VERSION` (e.g. `13.2`), else the toolkit's `cuda.h`
(also found under `targets/<platform>/include/`), else a probe of the resolved
binary. The result is clamped to the versions this crate can emit (13.2 to
13.3); a toolkit older than CUDA 13.2, or a probe that cannot run, is an error
rather than a silent fallback.

Set `CUTILE_TILEIRAS_PATH` to force a specific binary:

```bash
CUTILE_TILEIRAS_PATH=/opt/cuda-tile/bin/tileiras \
    cargo test -p cutile-compiler
```

Set `CUTILE_SETUP_DIAGNOSTICS=1` to print CUDA toolkit and `tileiras` discovery
decisions during setup.

## Testing

```bash
cargo test -p cutile-compiler
```

## Debugging

Set `CUTILE_DUMP` to inspect the compiler's output. Output goes to stderr,
once per compiled module.

```bash
# Dump the Tile IR for all kernels:
CUTILE_DUMP=ir cargo test -p cutile --test my_test -- --nocapture

# Dump both stages:
CUTILE_DUMP=ir,bytecode cargo test ...

# Dump everything (today: the same two stages):
CUTILE_DUMP=all cargo test ...
```

### Stages

| Stage | Description |
|-------|------------|
| `ir` | cutile-ir Module, pretty-printed (MLIR-like text) |
| `bytecode` / `bc` | Encoded bytecode, decoded to human-readable text |

The pass-level names `ast`, `resolved`, `typed`, and `instantiated` are
accepted but no pass emits them yet; they produce no output.

### Filtering

Use `CUTILE_DUMP_FILTER` to limit output. The two stages are module-level
dumps, so a qualified entry narrows by its module part (the function part is
not consulted), and a bare function name matches every module:

```bash
# By qualified path — dumps every kernel of `my_module`:
CUTILE_DUMP=ir CUTILE_DUMP_FILTER=my_module::my_kernel cargo test ...

# Multiple filters (comma-separated):
CUTILE_DUMP=ir CUTILE_DUMP_FILTER=my_module::add,other_module::gemm cargo test ...
```

### Legacy

`TILE_IR_DUMP=1` is still supported as an alias for `CUTILE_DUMP=ir`.
