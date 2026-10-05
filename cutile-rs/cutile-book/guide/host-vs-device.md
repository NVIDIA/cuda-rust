# Host vs. Device Code

cuTile Rust programs have two parts. Host code runs on the CPU and owns normal Rust control flow, tensor allocation, stream selection, kernel launch, and result readback. Device code runs on the GPU and describes the work performed by each tile block.

```rust
use cutile::prelude::*;
use kernels::scale;

fn main() -> Result<(), cuda_async::error::DeviceError> {
    let device = cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;

    let x = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
    let mut z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;

    let _ = scale((&mut z).partition([4, 4]), &x, 2.0f32).sync_on(&stream)?;
    Ok(())
}

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn scale<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        alpha: f32,
    ) {
        let tile_x = x.load_like(z);
        z.store(tile_x * alpha);
    }
}
```

`main` is host code. It constructs tensors, partitions the writable output, and synchronizes the returned `DeviceOp`. `scale` is device code. It runs once per tile block and sees one mutable output partition at a time.

## Modules and Entry Points

`#[cutile::module]` marks a Rust module whose functions can be compiled for the GPU. `#[cutile::entry()]` marks a function as a kernel entry point. Entry points are callable from host code through generated launcher APIs.

Entry points follow four rules:

1. They must be inside a `#[cutile::module]`.
2. Writable tensor parameters use static tile shapes, such as `Tensor<f32, S>` or `Tensor<f32, { [BM, BN] }>`.
3. Read-only tensor parameters may use dynamic dimensions, such as `Tensor<f32, { [-1, -1] }>`.
4. Kernels write results into tensor parameters instead of returning values.

Unmarked functions inside a `#[cutile::module]` are device functions. They can be called from entry points or other device functions and are inlined during compilation, but they cannot be launched directly.

## Kernel Launchers

The generated launcher accepts host-side values that correspond to the device-side kernel signature:

| Kernel parameter | Host input | Device view |
|---|---|---|
| `&Tensor<T, S>` | `&Tensor<T>`, `Arc<Tensor<T>>`, or `Tensor<T>` | Read-only tensor |
| `&mut Tensor<T, S>` | `Partition<&mut Tensor<T>>` or `Partition<Tensor<T>>` | Writable partition |
| Scalar (`f32`, `i32`, etc.) | Same scalar | Same scalar |

Mutable tensors are partitioned before launch so each tile block writes a disjoint region:

```rust
let mut z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;
let _ = scale((&mut z).partition([4, 4]), &x, 2.0f32).sync_on(&stream)?;
```

Read-only tensors can be borrowed, moved, or shared with `Arc`. Multiple tile blocks may read the same tensor concurrently.

## Host and Device Types

The same names appear on both sides, but host and device types carry different information:

| Type | Side | Parameterized by | Use |
|---|---|---|---|
| `Tensor<T>` | Host | Element type | GPU allocation managed from CPU code |
| `Partition<Tensor<T>>` | Host | Element type | Owned writable launch partition |
| `Partition<&mut Tensor<T>>` | Host | Element type | Borrowed writable launch partition |
| `Arc<Tensor<T>>` | Host | Element type | Shared read-only input |
| `Tensor<E, S>` | Device | Element type and shape | Kernel tensor parameter |
| `Partition<E, S>` | Device | Element type and shape | Device-side tiled view of a read-only tensor |
| `MappedPartitionMut<E, S, M>` | Device | Element type, tile shape, and map shape | Advanced writable output that lets one tile block process multiple logical output tiles |
| `Tile<E, S>` | Device | Element type and static shape | Register-resident compute value |

Host tensors use runtime shapes because allocation sizes are ordinary runtime data. Device tensors and tiles carry shape information in the type system so the compiler can check operations and specialize generated code.

`MappedPartitionMut` is used by kernels that need a custom traversal over the output grid, such as persistent GEMM. The host creates one by mapping a mutable partition:

```rust
let z = z.partition([BM, BN]).map([4, 1], num_tile_blocks);
```

The device entry point takes it by value:

```rust
fn kernel<const MAP_SHAPE: [i32; 2]>(
    mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
) {
    for out_idx in z.iter_indices() {
        // Compute one logical output tile.
        z.store(tile, out_idx);
    }
}
```

## DeviceOp Basics

Tensor constructors, kernel launchers, host readbacks, and composition helpers return `DeviceOp`s. A `DeviceOp` is a lazy description of GPU work. Nothing runs until it is synchronized, awaited, or captured into a CUDA graph.

```rust
let z = api::zeros::<f32>(&[32, 32]); // No allocation has run yet.
let z = z.sync_on(&stream)?;          // The operation runs here.
```

Kernel launchers are `DeviceOp`s too:

```rust
let op = scale((&mut z).partition([4, 4]), &x, 2.0f32);
let _ = op.sync_on(&stream)?;
```

The full host-side execution model is described in [Device Operations](device-operations.md).

## First-Use Compilation

Calling a generated launcher may compile the device code before it launches. A specialization is one compiled kernel variant for a particular entry function, target GPU, and set of compile-time inputs. The first launch of a specialization compiles the captured Rust AST to Tile IR bytecode and then to a cubin. Later launches with the same specialization reuse the cached binary.

Specialization depends on element types, const generic values, compile options, and other compile-time parameters. Dynamic tensor dimensions can vary across launches without creating a new specialization. [Compilation](jit-compilation.md) covers the cache and specialization rules in detail.

---

Continue to [Tensors and Tiles](tensors-and-tiles.md).
