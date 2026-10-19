# compiler2 — tile-ir Backend

compiler2 translates Rust AST into tile-ir operations and emits bytecode
without LLVM/MLIR. It handles type compilation and generic resolution.

## Bytecode Version: negotiated per toolchain (13.2 to 13.3)

The `cutile-ir` writer can emit bytecode versions 13.2 and 13.3
(`BytecodeVersion::MIN_SUPPORTED` to `BytecodeVersion::CURRENT`). The version
actually written is negotiated per toolchain by
`cuda_tile_runtime_utils::serialize_tile_ir_bytecode`: an explicit
`CUTILE_BYTECODE_VERSION` (e.g. `13.2`), else the resolved toolkit's `cuda.h`
`CUDA_VERSION`, else a probe of the resolved `tileiras`. The result is clamped
to the emittable range; a toolkit older than CUDA 13.2 (the Tile floor) or a
probe that cannot run is an error, not a fallback. 13.1 is not offered: the
writer's field layouts are the 13.2 ones (see below), and a "13.1" image with
those layouts was rejected by every toolkit.

### v13.2 Changes (vs v13.1)

| Op | Change | Details |
|----|--------|---------|
| `ForOp` | Added `unsignedCmp` flag bit | Bit 0 of flags field. Default 0. |
| `NegIOp` | Added `overflow` attribute | `IntegerOverflow` enum (varint). Default `NONE` (0). |
| `PrintTkoOp` | Added result token + token operand | Always 1 result (Token type). Flag bit 0 = has token operand. |
| `TanHOp` | Added `rounding_mode` attribute | `RoundingMode` enum (varint). Default `FULL` (5). |

### v13.3 Changes

| Op | Change | Details |
|----|--------|---------|
| `AllocaOp` | New op | Automatic allocation with `num_elem`, `alignment`, and `global` flag. |
| `AtomicRedViewTkoOp` | New op | View-based atomic reduction: result token + `view`, variadic `index`, `value`, optional `token`. |
| `ExpOp` | Added `rounding_mode` attribute | `RoundingMode` enum. Default `FULL` (5). |
| `LoadViewTkoOp` / `StoreViewTkoOp` | Index operands generalized | Gather/scatter views can use a 1D tensor index at the sparse dimension. |
| `MakeGatherScatterViewOp` | New op | Builds `gather_scatter_view` from a tensor view. |
| `MakeStridedViewOp` | New op | Builds `strided_view` from a tensor view. |
| `MmaFOp` | Added `fast_acc` flag bit | Bit 0 of flags field. Default 0. |
| `MmaFScaledOp` | New op | Result type + `lhs`, `rhs`, `acc`, `lhs_scale`, `rhs_scale`. |
| `PackOp` / `UnpackOp` | New ops | Rank-1 tile byte packing/unpacking for sub-byte types such as `f4E2M1FN`. |
| `GlobalOp` | Added `constant` flag + `symbol_visibility` attribute | Flag bit for constant; `SymbolVisibility` enum. Default `Public`. |
| `ModuleOp` | Added optional `producer` attribute | String attribute with flag bit. |

New 13.3 type tags covered by the writer/decoder include `f4E2M1FN`, `i4`,
`gather_scatter_view`, and `strided_view`; `f8E8M0FNU` is used as the FP8 scale
type for scaled MMA.

## Builder API Examples

### Constant (scalar)

```rust
use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::*;

let result_ty = Type::Tile(TileType {
    shape: vec![],
    element_type: TileElementType::Scalar(ScalarType::I32),
});
// i1 true = 0xFF, false = 0x00 (MLIR all-ones convention)
// Other integer types: little-endian bytes
let data = 42i32.to_le_bytes().to_vec();

let (op, results) = OpBuilder::new(Opcode::Constant, Location::Unknown)
    .attr("value", Attribute::DenseElements(DenseElements {
        element_type: result_ty.clone(),
        shape: vec![],
        data,
    }))
    .result(result_ty)
    .build(&mut module);
append_op(&mut module, block_id, op);
let value = results[0];
```

### Binary arithmetic (AddF with flags)

```rust
let tile_f32 = Type::Tile(TileType {
    shape: vec![128],
    element_type: TileElementType::Scalar(ScalarType::F32),
});

let (op, results) = OpBuilder::new(Opcode::AddF, Location::Unknown)
    .operand(lhs)
    .operand(rhs)
    .attr("rounding_mode", Attribute::i32(0)) // nearest_even
    .attr("flush_to_zero", Attribute::Bool(true))
    .result(tile_f32)
    .build(&mut module);
append_op(&mut module, block_id, op);
```

### Comparison (CmpI with signedness)

```rust
let tile_i1 = Type::Tile(TileType {
    shape: vec![],
    element_type: TileElementType::Scalar(ScalarType::I1),
});

let (op, results) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
    .operand(a)
    .operand(b)
    .attr("comparison_predicate", Attribute::i32(2)) // less_than
    .attr("signedness", Attribute::i32(1))            // signed
    .result(tile_i1)
    .build(&mut module);
append_op(&mut module, block_id, op);
```

### For loop (v13.2 with flags)

```rust
// Build the loop body region.
let (body_region, body_block, body_args) =
    build_single_block_region(&mut module, &[tile_i32.clone()]);
// body_args[0] is the induction variable

let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(&mut module);
append_op(&mut module, body_block, cont);

// Build the for op. v13.2 adds a flags field (unsignedCmp).
let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
    .operand(lower_bound)
    .operand(upper_bound)
    .operand(step)
    .region(body_region)
    .build(&mut module);
append_op(&mut module, block_id, for_op);
```

### Print (v13.2 — requires Token result)

```rust
let (op, results) = OpBuilder::new(Opcode::Print, Location::Unknown)
    .attr("str", Attribute::String("value = %i\n".into()))
    .attr("operandSegmentSizes", Attribute::Array(vec![
        Attribute::i32(1), // number of format args
        Attribute::i32(1), // 1 when an ordering token operand follows the args, else 0
    ]))
    .operand(value)
    .operand(token) // optional; sets flags bit 0 and is written after the sized args
    .result(Type::Token) // v13.2: PrintTkoOp always has 1 token result
    .build(&mut module);
append_op(&mut module, block_id, op);
```

### Load from pointer (LoadPtrTko with optional operands)

```rust
let tile_f32_128 = Type::Tile(TileType {
    shape: vec![128],
    element_type: TileElementType::Scalar(ScalarType::F32),
});

let mut op_builder = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
    .result(tile_f32_128)
    .result(Type::Token)
    .operand(source_ptr)          // group 0: source
    .operand(mask)                // group 1: mask (optional)
    .operand(padding)             // group 2: padding (optional)
    .attr("memory_ordering_semantics", Attribute::i32(0)) // weak
    .attr("operandSegmentSizes", Attribute::Array(vec![
        Attribute::i32(1), // source
        Attribute::i32(1), // mask
        Attribute::i32(1), // padding
        Attribute::i32(0), // token
    ]));
// Optional: memory_scope (only if not "weak")
// op_builder = op_builder.attr("memory_scope", Attribute::i32(0));

let (op, results) = op_builder.build(&mut module);
append_op(&mut module, block_id, op);
let loaded_tile = results[0];
let token = results[1];
```

### Entry function (top-level kernel)

```rust
let func_type = Type::Func(FuncType {
    inputs: vec![tile_f32.clone(), tile_f32.clone()],
    results: vec![],
});
let (region_id, block_id, args) =
    build_single_block_region(&mut module, &[tile_f32.clone(), tile_f32.clone()]);

// ... build ops in the block using args[0], args[1] ...

let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
append_op(&mut module, block_id, ret);

let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
    .attr("sym_name", Attribute::String("my_kernel".into()))
    .attr("function_type", Attribute::Type(func_type))
    .region(region_id)
    .build(&mut module);
module.functions.push(entry);
```

## Key Encoding Rules

- **i1 constants**: True = `0xFF`, False = `0x00` (MLIR all-ones convention).
- **Float attributes**: Encoded via `writeAPInt(bitcastToAPInt())` — signed varint for 16-64 bit, raw byte for 8-bit (F8).
- **DenseI32Array**: `varint(len) + len * LE_i32`. Used for `permutation`. (`operandSegmentSizes` is never serialized: it must be an `Attribute::Array` of `Attribute::i32`, and only drives how the writer groups operands.)
- **Inline vs self-contained attributes**: Op attributes are inline (no tag prefix). Array/Dictionary elements are self-contained (with tag).
- **Constant pool**: `varint(data_len) + raw_bytes`, referenced by pool index from ops.

## Reference Implementations

- **Python**: `cutile-python/src/cuda/tile/_bytecode/encodings.py` (per-op encoders)
- **C++ generated**: `target/release/build/cuda-tile-rs-*/out/build/lib/Bytecode/Writer/Bytecode.inc`
