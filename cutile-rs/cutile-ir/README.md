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

## Tile library API

Tile IR 13.5 (public preview `simt`) lets a Tile program call a SIMT
function from a *tile library*. `cutile_ir::tilelib` describes the two
bytecode halves of such a library; `tileiras --preview-create-tile-lib`
combines them with the code payload.

```text
                 interface (.nvtlifc)                 implementation catalog (.nvtl.0)
   ExternFunc ----------------------------> .tilebc    ExternImpl ----------------------> .tilebc
   @scale(%x: tile<128xf32> inout)                     @scale_impl implements @scale
     -> tile<128xf32>                                    (%x: fragment<1xf32, local,
                                                              mem_layout (1):(1),
   caller: preview$call @scale(%t)                            fragmap (128, 1, 1):(1, 0, 0)> inout)
            ^ resolved to @scale_impl by tileiras -l           ^ one invocation per tile element
```

```rust
use cutile_ir::ir::*;
use cutile_ir::tilelib::*;

let tile = TileType { shape: vec![128], element_type: TileElementType::Scalar(ScalarType::F32) };
let decl = ExternFunc::new("scale", vec![TileArg::new("x", tile, Access::InOut)]);
let fragment = FragmentType::new(
    vec![1], ScalarType::F32, MemorySpace::Local,
    Layout::contiguous_mem(1), Layout::elementwise(128),
);
let imp = ExternImpl::new("scale_impl", "scale", vec![FragmentArg::new("x", fragment, Access::InOut)]);

verify_library(&[decl.clone()], &[imp.clone()])?;
let interface = cutile_ir::write_bytecode(&interface_module("lib", &[decl]))?;
let catalog = cutile_ir::write_bytecode(&implementation_module("lib", &[imp]))?;
```

`write_bytecode` selects 13.5 and sets the preview header tag whenever a
module carries preview content. An assembler reads preview content only at
the version it calls current, the newest numbered version its
`--list-versions` prints, so a library meant for a newer assembler goes
through `write_bytecode_version` with that version (the CUDA 13.6
development assembler wants 13.6); asking for a version older than 13.5 is
an error, as in the reference writer. `Layout::elementwise`,
`warp_lanes`, `footprint_blocks` and `shared_whole` build the common fragmaps;
`to_mlir_text()` prints the modules in the textual form of the Tile IR
fixtures.

`ExternImpl::new` leaves every optional clause unset. `synchronization` is
an `Option`: `None` is an implementation without a `synchronization` clause,
and `Some(Synchronization::None)` writes the clause with its default value,
as the upstream scratch fixtures do. A cooperative implementation sets
`Some(Synchronization::Warp)` together with a fragmap such as
`Layout::warp_lanes(128)`, whose 32 participants satisfy the warp domain.

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
