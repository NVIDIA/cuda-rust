# 11. Inference with NVFP4/MXFP8

NVFP4 is an NVIDIA low-precision inference format that represents model values
with 4-bit E2M1 floating-point data and separate scale tensors. The FP4 values
are packed two per byte, so a model-loading layer needs to hand cuTile two kinds
of information:

- the packed FP4 bytes and their logical matrix shape
- the scale tensors and their layout

cuTile does not require a specific model-loading tool. The boundary is a typed
Rust value: packed FP4 bytes become `f4e2m1fnx2`, logical FP4 compute uses
`f4e2m1fn`, and scale tensors use an FP8 scale type such as `f8e4m3fn` or
`f8e8m0fnu`.

The examples use a linear layer with `mmaf_scaled` for matrix multiplication.
The same operation supports MXFP8 with FP8 data and E8M0 block scales.

---

## Why Block Scaling?

FP4 values are compact, but they have very limited range on their own. Block
scaling keeps the FP4 payload small while giving each fixed-size group of
logical FP4 values a separate scale factor. In the NVFP4 E4M3-scale layout used
in this tutorial, that ratio is one FP8 scale value per 16 logical FP4 values
along K. The matrix multiply consumes both pieces: the packed FP4 data carries
the low-precision values, and the scale tensors restore the range needed for
useful computation.

Sixteen `f16` values take 32 bytes. The
same 16 logical values in this NVFP4 layout take 8 packed FP4 bytes plus one FP8
scale byte, or 9 bytes total. That is about 28% of the `f16` operand storage,
or about 3.6x less operand bandwidth before accounting for output traffic.

For a `16 x 16 x 16` GEMM, the two `f16` operand tiles require 1024 bytes:

```text
A[16, 16] f16 + B[16, 16] f16 = 512 bytes + 512 bytes
```

The corresponding NVFP4 operands plus E4M3 scale values require 288 bytes:

```text
A packed FP4: 128 bytes + A scale values: 16 bytes
B packed FP4: 128 bytes + B scale values: 16 bytes
```

The 3.6x reduction applies to input operands. Outputs and accumulators still
use `f32`. Inference benefits when it repeatedly reads large activation and
weight matrices.

Conceptually, `mmaf_scaled` computes a matrix multiply where each operand value
is multiplied by the scale for its K-group:

```text
out[i, j] = acc[i, j] +
    sum over k:
        fp4_lhs[i, k] * lhs_scale[i, k / V] *
        fp4_rhs[k, j] * rhs_scale[k / V, j]
```

`V` is the scaling group size along K. Here `V = 16`, so a
`[BM, BK] @ [BK, BN]` tile uses `[BM, BK_SCALES]` left scale values and
`[BK_SCALES, BN]` right scale values, where `BK_SCALES = BK / 16`. With
`BK = 64`, that means four scale values along K. The call shape is:

- `lhs`: `Tile<f4e2m1fn, [BM, BK]>`
- `rhs`: `Tile<f4e2m1fn, [BK, BN]>`
- `lhs_scale`: `Tile<f8e4m3fn, [BM, BK_SCALES]>`
- `rhs_scale`: `Tile<f8e4m3fn, [BK_SCALES, BN]>`
- `acc`: `Tile<f32, [BM, BN]>`

```rust
let out = mmaf_scaled(lhs, rhs, acc, lhs_scale, rhs_scale);
```

---

## Storage and Compute Types

cuTile Rust separates packed storage from logical compute values:

| Rust type | Where it appears | Meaning |
|---|---|---|
| `f4e2m1fnx2` | `Tensor<f4e2m1fnx2, ...>` and `Tile<f4e2m1fnx2, ...>` | One byte containing two FP4 E2M1 values |
| `f4e2m1fn` | `Tile<f4e2m1fn, ...>` | One logical FP4 E2M1 value after unpacking |
| `f8e4m3fn` / `f8e8m0fnu` | `Tensor` or `Tile` scale values | FP8 block-scale formats used by scaled low-precision MMA |

At tensor boundaries, use `f4e2m1fnx2` for packed storage; `f4e2m1fn` appears
after unpacking inside the kernel. Each `f4e2m1fnx2` stores the first logical
FP4 value in the low nibble and the second logical FP4 value in the high
nibble. A nibble is one 4-bit half of a byte.

The constructors make the layout explicit:

```rust
use cutile::cuda_core::{f4e2m1fn, f4e2m1fnx2};

let byte = f4e2m1fnx2::from_bits(0xAB);
assert_eq!(byte.low(), f4e2m1fn(0x0B));
assert_eq!(byte.high(), f4e2m1fn(0x0A));

let pair = f4e2m1fnx2::from_nibbles(0x3, 0xC);
assert_eq!(pair.to_bits(), 0xC3);
```

---

## Inference Data Layout

For the simple linear layer in this tutorial, inference data is arranged like
this:

| Logical quantity | Storage type | Storage shape |
|---|---|---|
| activations `[M, K]` | `f4e2m1fnx2` | `[M, K / 2]` |
| weights `[N, K]` | `f4e2m1fnx2` | `[N, K / 2]` |
| activation scales | `f8e4m3fn` or `f8e8m0fnu` | `[M, K / 16]` for the layout used here |
| weight scales | `f8e4m3fn` or `f8e8m0fnu` | `[N, K / 16]` for the layout used here |
| output | `f32` | `[M, N]` |

cuTile reads `f4e2m1fnx2` values as already-packed bytes. The model loader must
put each pair in the correct nibble order.

This tutorial uses logical row-major scale tensors with the `V = 16` group size
described above. If your model-loading layer uses a different scale tensor
layout, adapt the bytes into the layout expected by your kernel before launching
it.

For a dense linear layer, the logical operation is:

```text
output[M, N] = alpha * activations[M, K] @ weights[N, K].T
```

`alpha` is an ordinary `f32` scalar. If your model data has global input or
weight scales, combine them into `alpha` before launch.

---

## Converting Host Bytes

The code below creates typed device tensors from plain byte vectors.

```rust
use cutile::api::{self, DeviceOpReshape};
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use std::sync::Arc;

fn packed_fp4(bytes: Vec<u8>) -> Arc<Vec<f4e2m1fnx2>> {
    Arc::new(bytes.into_iter().map(f4e2m1fnx2::from_bits).collect())
}

fn fp8_e4m3(bytes: Vec<u8>) -> Arc<Vec<f8e4m3fn>> {
    Arc::new(bytes.into_iter().map(f8e4m3fn).collect())
}

let m = 128usize;
let n = 256usize;
let k = 512usize;

// Self-contained example data. In a real model, these bytes come from the
// model-loading path after it has arranged data in the layout above.
let activation_bytes = vec![0x22u8; m * k / 2];
let weight_bytes = vec![0x22u8; n * k / 2];
let activation_scale_bytes = vec![0x38u8; m * k / 16];
let weight_scale_bytes = vec![0x38u8; n * k / 16];

let x = api::copy_host_vec_to_device(&packed_fp4(activation_bytes))
    .reshape(&[m, k / 2]);
let y = api::copy_host_vec_to_device(&packed_fp4(weight_bytes))
    .reshape(&[n, k / 2]);
let x_scales = api::copy_host_vec_to_device(&fp8_e4m3(activation_scale_bytes))
    .reshape(&[m, k / 16]);
let y_scales = api::copy_host_vec_to_device(&fp8_e4m3(weight_scale_bytes))
    .reshape(&[n, k / 16]);

let alpha = 1.0f32;
```

Convert each byte to packed FP4 storage:

```rust
u8 -> f4e2m1fnx2::from_bits(byte)
```

The conversion changes the type without changing the memory layout. To pack
FP4 values yourself, use `f4e2m1fnx2::from_nibbles(low, high)`.

### Tensor<u8> Escape Hatch

Prefer `Tensor<f4e2m1fnx2, ...>` for packed FP4 inputs because the kernel
signature then documents the data contract. If an interop boundary already owns
a `Tensor<u8>` and changing its type is inconvenient, cuTile still supports the
low-level path: load the bytes as `u8` and use generic `unpack` to produce
logical FP4 values.

```rust
#[cutile::entry()]
fn unpack_packed_fp4_bytes(
    output: &mut Tensor<f4e2m1fnx2, { [32] }>,
    input: &Tensor<u8, { [-1] }>,
) {
    let bytes = load_tile(input, shape![32], [0]);
    let f4s: Tile<f4e2m1fn, { [64] }> = unpack(bytes);
    let packed = f4s.pack(shape![32]);
    output.store(packed);
}
```

Every `u8` bit pattern is valid byte storage, so this unpacking is safe.

---

## Kernel Pattern

The kernel loads packed bytes, unpacks them to logical FP4 matrix tiles, then
calls `mmaf_scaled`:

```rust
#[cutile::module]
mod nvfp4_linear {
    use cutile::core::*;

    #[cutile::entry()]
    fn linear_tile<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BK_PACKED: i32,
        const BK_SCALES: i32,
    >(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        y: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        y_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid_m = program_id(0);
        let pid_n = program_id(1);

        let part_x = x.partition(shape![BM, BK_PACKED]);
        let part_y = y.partition(shape![BN, BK_PACKED]);
        let part_x_scales = x_scales.partition(shape![BM, BK_SCALES]);
        let part_y_scales = y_scales.partition(shape![BN, BK_SCALES]);

        let mut tile_z = constant(0.0f32, shape![BM, BN]);

        for k_tile in 0i32..num_tiles(&part_x, 1) {
            let tile_x_packed = part_x.load([pid_m, k_tile]);
            let tile_y_packed = part_y.load([pid_n, k_tile]);

            let tile_x = tile_x_packed.unpack(shape![BM, BK]);
            let tile_y = tile_y_packed.unpack(shape![BN, BK]).transpose();

            let tile_x_scales = part_x_scales.load([pid_m, k_tile]);
            let tile_y_scales = part_y_scales.load([pid_n, k_tile]).transpose();

            tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
        }

        let alpha_tile = broadcast_scalar(alpha, z.shape());
        z.store(tile_z * alpha_tile);
    }
}
```

Unpacking doubles the number of elements:

```text
Tile<f4e2m1fnx2, [BM, BK_PACKED]>  // BK_PACKED = BK / 2
    unpack(shape![BM, BK]) -> Tile<f4e2m1fn, [BM, BK]>
```

Do not pass `Tile<f4e2m1fnx2, ...>` directly to `mmaf_scaled`. The MMA operand
is the unpacked logical `Tile<f4e2m1fn, ...>`. The method emits the required
rank-1 Tile IR `unpack` internally and reshapes the result back to the requested
tile shape.

---

## Launch Shape

For the kernel above, `BK` is the logical K tile width. The kernel walks the K
tiles of the packed activation partition with `num_tiles(&part_x, 1)`; the host
passes `BK` plus the derived packed and scale tile widths used in static tile
shapes:

```rust
use cutile::api;
use cutile::tile_kernel::TileKernel;
use cutile::tensor::IntoPartition;

let bm = 16usize;
let bn = 16usize;
let bk = 64usize;
let bk_packed = bk / 2;
let bk_scales = bk / 16;
let z = api::zeros::<f32>(&[m, n]).partition([bm, bn]);

let (z, ..) = nvfp4_linear::linear_tile(
    z,
    x,
    y,
    x_scales,
    y_scales,
    alpha,
)
.generics(vec![
    bm.to_string(),
    bn.to_string(),
    bk.to_string(),
    bk_packed.to_string(),
    bk_scales.to_string(),
])
.sync()?;
```

---

## Architecture Tradeoffs

Packing FP4 values reduces operand bytes before the data reaches the kernel.
That can reduce memory traffic whenever the model is stored and moved in packed
form. The larger speedup comes when the target GPU also has hardware support for
block-scaled FP4 matrix multiply, because the compiler can lower
`mmaf_scaled` on logical `f4e2m1fn` tiles to NVFP4-specific Tensor Core
instructions.

CUDA Tile IR 13.3 supports `sm_80` and newer targets generally, but that does
not mean every Tile IR type or MMA operation is native on every target.
`mmaf_scaled` with scaled floating-point inputs is an `sm_100` and newer
operation, and `f4E2M1FN` is a Blackwell-supported Tile IR type.

| Target family | Native `f4E2M1FN` type | Specialized scaled FP4 MMA |
|---|---|---|
| `sm_80` / Ampere | No | No |
| `sm_90` / Hopper | No | No |
| `sm_100` / Blackwell | Yes | Yes, through `mmaf_scaled` lowering to NVFP4 Tensor Core instructions |
| `sm_120` / Blackwell | Yes | Yes, through `mmaf_scaled` lowering to NVFP4 Tensor Core instructions |

Targets without native support usually need to unpack or convert FP4 before
the GEMM. That work can offset the bandwidth savings. Benchmark the fallback
against the established implementation for the target GPU.

The runnable example in the repository is `cutile-examples/examples/nvfp4.rs`.
On native NVFP4 targets, it checks the kernel output against an all-ones input
case.

---

## MXFP8-Style FP8

The FP8 version uses the same `mmaf_scaled` operation but does not need
sub-byte packing or unpacking. The data tensors store one FP8 value per byte,
and the scale tensors use one E8M0 scale value per 32 FP8 values along K:

| Logical quantity | Storage type | Storage shape |
|---|---|---|
| activations `[M, K]` | `f8e4m3fn` | `[M, K]` |
| weights `[N, K]` | `f8e4m3fn` | `[N, K]` |
| activation scales | `f8e8m0fnu` | `[M, K / 32]` |
| weight scales | `f8e8m0fnu` | `[N, K / 32]` |
| output | `f32` | `[M, N]` |

Inside the kernel, load the FP8 tiles directly, load the E8M0 scale tiles, and
call `mmaf_scaled`:

```rust
let tile_x = part_x.load([pid_m, k_tile]);
let tile_y = part_y.load([pid_n, k_tile]).transpose();
let tile_x_scales = part_x_scales.load([pid_m, k_tile]);
let tile_y_scales = part_y_scales.load([pid_n, k_tile]).transpose();

tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
```

The main shape differences from NVFP4 are:

```text
BK_PACKED is not needed
BK_SCALES = BK / 32

lhs:       Tile<f8e4m3fn,  [BM, BK]>
rhs:       Tile<f8e4m3fn,  [BK, BN]>
lhs_scale: Tile<f8e8m0fnu, [BM, BK_SCALES]>
rhs_scale: Tile<f8e8m0fnu, [BK_SCALES, BN]>
```

The runnable example in the repository is `cutile-examples/examples/mxfp8.rs`.
On native MXFP8 targets, it checks the kernel output against an all-ones input
case.

For the operation-level reference behind this pattern, see
[Matrix Multiply](../reference/dsl-api.md#matrix-multiply).
