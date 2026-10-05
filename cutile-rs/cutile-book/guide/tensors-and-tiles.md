# Tensors and Tiles

cuTile Rust kernels operate by moving data from tensors into tiles, computing on those tiles, and storing the result back to tensors.

| Property | Tensor | Tile |
|---|---|---|
| Location | Global memory (HBM) | Registers |
| Mutability | Mutable or read-only | Immutable |
| Shape | Static, dynamic, or mixed | Static |
| Operations | Load and store | Arithmetic, reductions, matrix multiply, shape ops |
| Lifetime | Persists across kernels | Exists only inside a kernel |
| Addressable | Yes | No |

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,
    x: &Tensor<f32, { [-1, -1] }>,
    y: &Tensor<f32, { [-1, -1] }>,
) {
    let tile_x = x.load_like(z);
    let tile_y = y.load_like(z);
    z.store(tile_x + tile_y);
}
```

The output tensor is mutable and already partitioned by the host launcher. The input tensors are read-only and can be loaded into tiles that match the output partition.

## Tensor, Partition, Tile

`Tensor<E, S>` is the device-side view of global memory. Kernels receive tensors as parameters. A tensor can be loaded into a `Tile<E, S>` or used to create a device-side partition.

`Partition<E, S>` is a tiled view of a tensor. On the host, partitioning a mutable tensor determines how many tile blocks launch and which region each block writes. On the device, partitioning a read-only tensor lets a kernel load arbitrary tiles by index.

`Tile<E, S>` is an immutable register-resident array fragment. Tile operations create new tiles instead of mutating the original value:

```rust
let tile = x.load_like(z);
let shifted = tile + 1.0f32;
let scaled = shifted * 2.0f32;
z.store(scaled);
```

![Data flow: Load from Tensor to Tile, Compute in registers, Store back to Tensor](../_static/images/data-flow.svg)

```text
Tensor -> Partition -> Tile -> Compute -> Store
```

## Partitioning and the Grid

Mutable outputs are partitioned on the host before launch:

```rust
let mut z = api::zeros::<f32>(&[1024, 1024]).sync_on(&stream)?;
let _ = add((&mut z).partition([64, 64]), &x, &y).sync_on(&stream)?;
```

The partition shape becomes the static shape seen by the kernel. A `[1024, 1024]` tensor partitioned as `[64, 64]` creates a logical grid of `(16, 16, 1)` tile blocks. Each tile block receives one writable sub-tensor.

Read-only inputs do not need host-side partitioning. Inside the kernel, partition them with the shape needed by the algorithm:

```rust
let pid_m = program_id(0);
let part_x = x.partition(shape![BM, BK]);
let tile_x = part_x.load([pid_m, i]);
```

The same read-only tensor can be partitioned multiple ways inside one kernel. This is common in matrix multiplication, where the left-hand side and right-hand side are loaded with different tile shapes.

The launch grid is inferred from mutable output partitions unless the launcher sets it explicitly:

```rust
kernel(z.partition([64, 64]), x).grid((16, 16, 1)).sync_on(&stream)?;
```

When a kernel has multiple mutable outputs, their inferred grids must match.

## Static and Dynamic Shapes

Static dimensions are compile-time constants. Dynamic dimensions use `-1` and are resolved from the runtime tensor shape.

```rust
#[cutile::entry()]
fn normalize<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,          // Static tile shape from partition.
    x: &Tensor<f32, { [-1, -1] }>,   // Runtime full tensor shape.
) {
    let tile = x.load_like(z);
    z.store(tile);
}
```

Static shapes let the compiler check operations and optimize layout. Dynamic dimensions let the same compiled variant handle different full tensor sizes. The common pattern is static output tile shape and dynamic read-only input shape.

Const generic arrays such as `const S: [i32; 2]` and `shape![BM, BK]` carry tile dimensions through the type system. Changing a const generic value can create a new compiled variant.

## Loading, Computing, Storing

`input.load_like(output)` loads a read-only tensor region matching the output tensor's tile shape and tile-block coordinates. For explicit device-side partitions, call `partition.load(index)`:

```rust
let part_x = x.partition(shape![BM, BK]);
let tile_x = part_x.load([pid_m, k_tile]);
```

Writable tensors store tile results:

```rust
let result = tile_x + tile_y;
z.store(result);
```

Use `load_tile_mut` when a kernel needs to read the existing output value before writing a new one:

```rust
let acc = load_tile_mut(z);
z.store(acc + update);
```

(operations-at-a-glance)=
## Kernel operations

| Category | Examples |
|---|---|
| Load and store | `load_like` (method form of `load_tile_like`), `load_tile_mut`, `Partition::load`, `Tensor::store` |
| Arithmetic | `+`, `-`, `*`, `/`, `fma`, `true_div` |
| Math | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` |
| Reduction and scan | `reduce_max`, `reduce_sum`, `reduce_min`, `scan` |
| Matrix multiply | `mma`, `mmaf_scaled` |
| Shape manipulation | `reshape`, `broadcast`, `transpose`, `shape!` |
| Comparison | `gt_tile`, `ge_tile`, `lt_tile`, `le_tile`, `eq_tile`, `select` |
| Creation and conversion | `constant`, `iota`, `convert_tile`, `pack`, `unpack` |

For element types, operation signatures, and lower-level memory operations, see the [DSL API](../reference/dsl-api.md).

## Broadcasting and Reductions

Broadcasting expands a scalar or smaller tile to match a larger tile shape. It follows NumPy-style rules: align dimensions from the right; each dimension must either match or be 1.

```rust
let bias: Tile<f32, { [1, BN] }> = ...;
let x: Tile<f32, { [BM, BN] }> = ...;
let y = x + bias.broadcast(shape![BM, BN]);
```

Reductions collapse one axis. Reshape the reduced result before broadcasting it back:

```rust
fn softmax<const BM: i32, const BN: i32>(
    x: Tile<f32, { [BM, BN] }>,
) -> Tile<f32, { [BM, BN] }> {
    let max = reduce_max(x, 1i32)
        .reshape(shape![BM, 1])
        .broadcast(shape![BM, BN]);
    let stable = x - max;

    let exp_x = exp(stable);
    let sum = reduce_sum(exp_x, 1i32)
        .reshape(shape![BM, 1])
        .broadcast(shape![BM, BN]);

    true_div(exp_x, sum)
}
```

### Numerically Stable Softmax

Subtract the per-row maximum before `exp` when implementing softmax. This prevents overflow on large positive inputs and is the pattern used in fused softmax and attention kernels.

## Tiled Matrix Multiply

Matrix multiply accumulates repeated `mma` calls across the K dimension. Each tile block owns one output tile. Each loop iteration loads a `[BM, BK]` tile from the left input and a `[BK, BN]` tile from the right input.

```rust
fn tiled_gemm<
    E: ElementType,
    const BM: i32,
    const BN: i32,
    const BK: i32,
>(
    z: &mut Tensor<f32, { [BM, BN] }>,
    x: &Tensor<E, { [-1, -1] }>,
    y: &Tensor<E, { [-1, -1] }>,
) {
    let pid_m = program_id(0);
    let pid_n = program_id(1);
    let k_tiles = x.shape()[1] / BK;

    let part_x = x.partition(shape![BM, BK]);
    let part_y = y.partition(shape![BK, BN]);

    let mut acc = constant(0.0f32, shape![BM, BN]);
    for k_tile in 0i32..k_tiles {
        let tile_x = part_x.load([pid_m, k_tile]);
        let tile_y = part_y.load([k_tile, pid_n]);
        acc = mma(tile_x, tile_y, acc);
    }

    z.store(acc);
}
```

The output accumulator usually uses a wider type than the inputs. For example, FP16 or FP8 inputs often accumulate into FP32.

## Type Safety and Generics

The compiler catches shape mismatches, element-type mismatches, and invalid matrix multiply shapes before code runs:

```rust
let a: Tile<f32, { [16, 8] }> = ...;
let b: Tile<f32, { [16, 32] }> = ...;
let c = mma(a, b, acc); // Error: inner dimensions do not match.
```

Use explicit conversion when element types differ:

```rust
let y_float: Tile<f32, { [4, 4] }> = convert_tile(y_int);
let z = x_float + y_float;
```

Generic kernels can support multiple shapes and element types:

```rust
#[cutile::entry()]
fn scale<E: ElementType, const S: [i32; 2]>(
    z: &mut Tensor<E, S>,
    x: &Tensor<E, { [-1, -1] }>,
    alpha: E,
) {
    let tile = x.load_like(z);
    z.store(tile * alpha);
}
```

Each concrete element type and const generic value can produce a separate compiled variant. Dynamic tensor dimensions do not.

---

Continue to [Compilation](jit-compilation.md).
