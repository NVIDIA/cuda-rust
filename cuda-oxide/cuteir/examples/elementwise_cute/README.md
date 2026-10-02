# Elementwise add with Tensor tiles

This is the smallest `cute-rs` example:

```text
C[i] = A[i] + B[i]
```

It runs once with `f32` values and once with `f16` values. Each GPU thread owns
one contiguous 16-byte tile.

`Tensor` is a read-only view. `TensorMut` is a writable view that cannot be
copied or cloned, which makes accidental overlapping output owners harder to
create.

## The Tensor flow

`zipped_divide` gives one flat vector two coordinates:

```text
flat vector
[ 0  1  2  3 | 4  5  6  7 | 8  9 10 11 | ... ]

                zipped_divide::<4>()
                           │
                           ▼
coordinate = (place inside a tile, tile number)
```

The thread number selects the second coordinate. Loading the resulting slice
moves one tile into registers; storing moves the result back to global memory.

```text
A storage ──► Tensor ──► tiled view ──► slice(thread)
                                              │
                                              ▼
                                         load registers
                                              │
B storage ──► Tensor ──► tiled view ──► slice(thread)
                                              │
                                              ▼
                                         load registers
                                              │
                                      elementwise add
                                              │
                                              ▼
C storage ◄── TensorMut ◄── tiled view ◄── store registers
```

The Rust reads like the diagram:

```rust
let g_a = Tensor::from_slice(a).zipped_divide::<TILE>();
let g_b = Tensor::from_slice(b).zipped_divide::<TILE>();
let g_c = TensorMut::from_disjoint_slice(&mut c).zipped_divide::<TILE>();

let t_a = g_a.slice(tid);
let t_b = g_b.slice(tid);
let mut t_c = g_c.slice(tid);

let result = unsafe { t_a.load() } + unsafe { t_b.load() };
unsafe { t_c.store(result) };
```

## The final partial tile

The example sizes are intentionally not exact multiples of a tile:

```text
f32: 1003 values, 4 per tile

... | 996 997 998 999 | 1000 1001 1002  -- |
                         └─ scalar tail ────┘
```

Complete tiles use the tile load/add/store path. The final thread uses a small
Rust loop and touches only valid values. The `f16` run does the same thing with
eight values per tile.

## Run it

You need the normal CUDA Oxide toolchain and an NVIDIA GPU supported by the
project. From the `cuda-oxide/` directory:

```bash
cargo oxide run elementwise_cute
```

A successful run confirms:

- all 1,003 `f32` result bit patterns match the host reference;
- all 2,005 `f16` result bit patterns match the host reference;
- both vector-tile paths and their scalar tails are exercised.
