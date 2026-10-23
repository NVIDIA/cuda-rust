# Layouts, Tiling, and Partitions by Example

cute-layout and cute-rs let you describe how logical coordinates map to physical memory: whether that is a 1D vector split across threads, a row-wise reduction across warps, or a 2D swizzled shared-memory tile shared by host TMA encoding and device MMA.

---

## 1. Evaluating Layouts with cute-layout

cute-layout is a `no_std` crate shared by both cute-rs and the compiler. A `Layout` pairs a shape `IntTuple` with a stride `IntTuple` and maps logical coordinates to linear offsets:

```rust
use cute_layout::{IntTuple, Layout};

// Shape (2, 3) with row-major stride (3, 1)
let layout: Layout = "(2,3):(3,1)".parse().unwrap();
assert_eq!(layout.checked_call(&IntTuple::Leaf(1)), Some(3));
```

In kernel signatures, you write static layouts as Rust type markers (`C<N>`, Rust tuples such as `(A, B)` for nested modes, `Layout<Shape, Stride>`, `RowMajor<R, C>`, `ColMajor<R, C>`, `Swizzle<B, M, S>`, and `ComposedLayout<Swizzle, Offset, Layout>`).

---

## 2. 1D Tiling and Ragged Tails (zipped_divide and slice)

`zipped_divide::<TILE>()` restructures a flat 1D tensor into a 2D view indexed by `(offset_inside_tile, tile_number)`. Calling `.slice(tid)` selects the tile owned by thread `tid`. `local_tile::<TILE>(tile)` is the one-call form, equal to `zipped_divide::<TILE>().slice(tile)`: CuTe's `local_tile` for a one-dimensional tiler.

```{figure} images/zipped-divide-and-slice.svg
:align: center
:width: 100%

`zipped_divide::<4>()` and `.slice(tid)` partition a 1,003-element vector across 251 threads: threads `0..249` take the vectorized `cute.copy` path, while thread `250` runs a scalar loop over the 3-element tail.
```

```rust
let tid = thread::index_1d().get();
let g_a = cute_rs::Tensor::from_slice(a).zipped_divide::<4>();
let g_b = cute_rs::Tensor::from_slice(b).zipped_divide::<4>();
let g_c = cute_rs::TensorMut::from_disjoint_slice(&mut c).zipped_divide::<4>();

let t_a = g_a.slice(tid);
let t_b = g_b.slice(tid);
let mut t_c = g_c.slice(tid);

if t_a.is_full() {
    let sum = unsafe { t_a.load() } + unsafe { t_b.load() };
    unsafe { t_c.store(sum) };
} else {
    let mut k = t_a.base();
    while k < a.len() {
        unsafe { t_c.store_at(k, a[k] + b[k]) };
        k += 1;
    }
}
```

---

## 3. Shared Host/Device TMA Layout Types

For 2D and 3D Tensor Memory Accelerator (TMA) transfers, you define the swizzled shared-memory layout type **once** and use it on both the host and the device:

```{figure} images/host-device-tma-type-agreement.svg
:align: center
:width: 100%

A single `ComposedLayout<Swizzle, Offset, Layout>` type alias binds host `create_tensor_map_tiled` encoding to device `&TensorMap<T, Layout>` and `Sm100SharedTile`.
```

```rust
use cute_rs::copy::tma::{TmaEncodeOptions, create_tensor_map_tiled_with_options};
use cute_rs::{ComposedLayout, RowMajor, Swizzle, TensorMap};

// Shared layout type: 128x64 tile with a 128-byte swizzle
pub type ASmem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;

// 1. Host: encode the 128-byte hardware CUtensorMap from `ASmem`
let a_desc: TensorMap<f16, ASmem> = create_tensor_map_tiled_with_options(
    a_dev.as_device_ptr(),
    m,
    k,
    k,
    TmaEncodeOptions::default(),
)?;

// 2. Device: receive the descriptor by value via #[grid_constant]
#[kernel]
pub fn gemm_kernel(#[grid_constant] a: &TensorMap<f16, ASmem>) {
    // `a` is guaranteed to carry the exact `ASmem` tile shape and swizzle
}
```

If a kernel attempts to pass `&TensorMap<f16, ASmem>` into a copy or MMA expecting a different layout or element type, rustc rejects the program at compile time before device codegen even runs.
