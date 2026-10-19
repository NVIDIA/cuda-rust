# cutile-ir

Pure Rust IR builder and bytecode writer for the CUDA Tile dialect. Builds
Tile IR programs in-memory and serializes them to the bytecode format consumed
by `tileiras`. Builds with Cargo without an LLVM or C++ toolchain.

## Example

```rust
use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{write_bytecode, Opcode};
use cutile_ir::ir::*;

let mut m = Module::new("example");
let f32_ty = Type::Scalar(ScalarType::F32);

// Function body: block with two f32 args.
let (region, block, args) =
    build_single_block_region(&mut m, &[f32_ty.clone(), f32_ty.clone()]);

// addf %0, %1
let (add, res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
    .operand(args[0]).operand(args[1])
    .attr("rounding_mode", Attribute::i32(0))
    .result(f32_ty.clone())
    .build(&mut m);
append_op(&mut m, block, add);

// return %2
let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
    .operand(res[0])
    .build(&mut m);
append_op(&mut m, block, ret);

// Entry function wrapping the body region.
let func_type = FuncType {
    inputs: vec![f32_ty.clone(), f32_ty.clone()],
    results: vec![f32_ty],
};
let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
    .attr("sym_name", Attribute::String("add_kernel".into()))
    .attr("function_type", Attribute::Type(Type::Func(func_type)))
    .region(region)
    .build(&mut m);
m.functions.push(entry);

let bytecode = write_bytecode(&m).expect("serialization failed");
println!("{}", m.to_mlir_text());
```

See [`examples/build_basic.rs`](examples/build_basic.rs) for a fuller example
with tensor views, control flow, and permute operations.

## Builder API

Programs are built bottom-up: allocate blocks, fill them with operations,
wrap blocks in regions, attach regions to control-flow ops, register the
entry function in the module.

### Module

Root container. Owns all IR objects in flat arenas and holds the list of
entry functions. Values, operations, blocks, and regions are referenced by
lightweight index handles (`Value`, `OpId`, `BlockId`, `RegionId`) — no
lifetime parameters.

```rust
let mut module = Module::new("my_kernel");
// ... build IR ...
module.functions.push(entry_op_id);
```

### OpBuilder

Constructs a single operation via fluent API — chain `.operand()`,
`.result()`, `.attr()`, `.region()`, then `.build(&mut module)`.

```rust
let (op_id, results) = OpBuilder::new(Opcode::AddF, Location::Unknown)
    .operand(a).operand(b)
    .attr("rounding_mode", Attribute::i32(0))
    .result(Type::Scalar(ScalarType::F32))
    .build(&mut module);
```

Returns `(OpId, Vec<Value>)`. The op is allocated but not yet placed in a
block — call `append_op` next.

### append_op

Places an operation at the end of a block.

```rust
append_op(&mut module, block_id, op_id);
```

### build_single_block_region

Creates a region containing one block with typed arguments. Returns handles
for all three. This is the common case for function bodies, loop bodies,
and if/else branches.

```rust
let (region_id, block_id, args) =
    build_single_block_region(&mut module, &[Type::Scalar(ScalarType::I32)]);
// args[0] is the i32 block argument.
```

### Bytecode serialization

```rust
use cutile_ir::bytecode::write_bytecode_to_file;
write_bytecode_to_file(&module, "kernel.bc")?;
// Then: tileiras --gpu-name sm_120 -o kernel.cubin kernel.bc
```

`write_bytecode_to_file` uses the writer's current format, which may be newer
than an installed assembler. Use `write_bytecode_version` when targeting an
older format. The JIT, `build_basic` example and assembler-validation tests
negotiate with the selected executable; they also handle older assemblers
without `--list-versions`. See the [compatibility matrix](../cutile-book/reference/compatibility.md).

cuTile's JIT resolves `tileiras` from `CUTILE_TILEIRAS_PATH` when set, then
`$CUDA_TOOLKIT_PATH/bin/tileiras`, then the default CUDA 13.2+ install
directories (`/usr/local/cuda-13.4` through `/usr/local/cuda`), and only then
from `PATH`. Set `CUTILE_TILEIRAS_PATH` to force a specific binary, for
example:

```bash
CUTILE_TILEIRAS_PATH=/opt/cuda-tile/bin/tileiras \
    cargo test -p cutile-compiler
```

## Design

### Arena ownership

All IR objects (operations, blocks, regions, values) live in `Vec` arenas
owned by a single `Module`. References are `Copy` index handles with no
lifetime parameters, so IR fragments can be built out of order and stored
freely.

### melior-inspired API

The builder API mirrors melior's `OperationBuilder` pattern for easy porting:

| melior | cutile-ir |
|--------|-----------|
| `OperationBuilder::new("cuda_tile.addf", loc)` | `OpBuilder::new(Opcode::AddF, loc)` |
| `.add_operands(&[a, b])` | `.operand(a).operand(b)` |
| `.add_results(&[ty])` | `.result(ty)` |
| `.add_attributes(&[(id, attr)])` | `.attr("name", Attribute::X)` |
| `.build()?` | `.build(&mut module)` returns `(OpId, Vec<Value>)` |
| `block.append_operation(op)` | `append_op(&mut module, block_id, op_id)` |

Operations use a Rust `Opcode` enum instead of strings — typos are caught at
compile time.

### Direct bytecode emission

Writes the same binary format that the C++ `BytecodeWriter.cpp` produces.
`tileiras` accepts it without modification.

## Background

The cuTile Rust compiler originally used
[melior](https://github.com/edgl/melior) (thanks to
[Yota Toyama](https://github.com/raviqqe) for that project) to construct
MLIR operations in the CUDA Tile dialect. `cutile-ir` replaces the
LLVM/MLIR dependency with a Rust crate whose index-based API needs no
lifetime parameters.

## License

Apache-2.0
